define([
    './properties',
    'text!./hello-world.ng.html',
    'css!./hello-world.ng.css'
  ],
  function (props, template, style) {
    'use strict';

    // function scape(value, tid) {
    //   var card = angular.element.find(`[tid='${tid}']`)[0];

    //   var button = `
    //   <div class="qs-popover-container qs-toolbar__element extra-button">
    //     <button type="button" class="lui-button" tid="_007" qva-outside-ignore-for="nav-bookmark" title="">
    //       <span class="lui-icon lui-icon--drop qs-no-margin" aria-hidden="true"></span>
    //     </button>
    //   </div>`;

    //   // var createButton = false ? angular.element.find(`[tid='_007']`) : true;

    //   // console.log(createButton);

    //   // if (createButton) {
    //     var group = angular.element.find(`[class='qs-toolbar__right']`)[0];
    //     angular.element(group).prepend(angular.element(button));
    //   // }

    //   if (value) {
    //     angular.element(card).addClass("scape");
    //   } else {
    //     angular.element(card).removeClass("scape");
    //   }
    // }

    return {
      support: {
        export: true,
        exportData: true,
        snapshot: true,
      },
      definition: props,
      template: template,
      style: style,
      controller: ['$scope', '$compile', function ($scope, $compile) {
        // $scope.component.model.Validated.bind(function () {
        //   scape($scope.layout.props.scape, $scope.layout.qInfo.qId);
        // });

        angular.element(`#qs-toolbar-container`).addClass('cubricks-navbar');
        angular.element(`#qs-toolbar-container`).addClass('navbar-cubricks');

        $scope.scape = function () {
          console.log(' TEMP ');
        }

        // scape($scope.layout.props.scape, $scope.layout.qInfo.qId);
        // console.log($scope.layout);

        var navbar = angular.element(`<div id="cubricks-navbar" class="navbar"></div>`);

        // var btn = angular.copy(angular.element(`[tid='2f7a7e']`));
        // navbar.append($compile(btn)($scope));


        // var myClone = element.clone();
        // myClone[0].innerHTML = element[0].srcHTML;
        // // $compile(myClone)($scope);

        // navbar.append( $compile(myClone)($scope));

        //  var pElement = angular.element(`[tid='2f7a7e']`);
        //  var pElementCopy = pElement.clone();
        //  pElementCopy.html("Foo");

        //  $compile(pElementCopy)($scope);
        //  navbar.append(pElementCopy);


        // navbar.append(angular.element(`[tid='2f7a7e']`));
        // navbar.append(angular.element(`[tid='4dd782']`));


        navbar.append(angular.element(`
        <div id="cubricks-grroup-btn" class="lui-buttongroup">
          <lui-button ng-click="qlik.navigation.nextSheet()">Next sheet</lui-button>
          <lui-button ng-click="qlik.navigation.prevSheet()">Previous sheet</lui-button>
          <lui-button tid='4dd782'>Edit mode</lui-button>
          <lui-button ng-click="scape()">Edit mode</lui-button>
          <button type="button" class="lui-button lui-buttongroup__button" tid="4dd782" title="Edit sheet">
            <span class="lui-icon lui-icon--edit lui-button__icon" aria-hidden="true"></span>
          </button>
        </div>`));


        var container = angular.element('#qs-toolbar-container');
        container.append($compile(navbar)($scope));



        $scope.$on('$destroy', function () {

          angular.forEach($("[class*='cubricks']"), (element) => {
            angular.element(element).removeClass((_, classes) => {
              return classes.split(' ').filter((w) => w.includes('cubricks')).join(' ');
            });
          });

          angular.forEach($("[id*='cubricks']"), (element) => {
            angular.element(element).remove();
          });

        });

      }]
    };
  });