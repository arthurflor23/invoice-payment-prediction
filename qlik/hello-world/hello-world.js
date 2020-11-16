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
      definition: props,
      template: template,
      style: style,
      controller: ['$scope', function ($scope) {
        // $scope.component.model.Validated.bind(function () {
        //   scape($scope.layout.props.scape, $scope.layout.qInfo.qId);
        // });

        // $scope.component.model.Validated.bind(function () {
        //   scape($scope.layout.props.scape, $scope.layout.qInfo.qId);
        // });

        // scape($scope.layout.props.scape, $scope.layout.qInfo.qId);

        console.log($scope.layout);

        var container = angular.element('#qv-page-container');
        console.log(container);


        var navbar = angular.element(`<div id="cubricks-navbar" class="navbar"></div>`);

        navbar.append(angular.element(`[tid='2f7a7e']`))
        container.append(navbar);

        $scope.$on('$destroy', function (event) {
          angular.element('#cubricks-navbar').remove()
        });

      }]
    };
  });